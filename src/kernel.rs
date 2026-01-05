use cuda_sys::cuda;
use std::ffi::CString;

// Type aliases for clarity
type CUresult = cuda_sys::cuda::CUresult;
type CUmodule = cuda_sys::cuda::CUmodule;
type CUfunction = cuda_sys::cuda::CUfunction;

pub fn run_vector_add(a: &[f32], b: &[f32]) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let n = a.len();
    
    unsafe {
        // Initialize CUDA - use cuInit instead of cudaSetDevice
        let mut result = cuda::cuInit(0);
        check_error(result, "cuInit")?;
        
        // Get device
        let mut device: cuda_sys::cuda::CUdevice = 0;
        result = cuda::cuDeviceGet(&mut device, 0);
        check_error(result, "cuDeviceGet")?;
        
        // Create context
        let mut context: cuda_sys::cuda::CUcontext = std::ptr::null_mut();
        result = cuda::cuCtxCreate_v2(&mut context, 0, device);
        check_error(result, "cuCtxCreate")?;
        
        // Load module from PTX string
        let ptx_cstr = CString::new(include_str!("../kernel.ptx"))?;

        let mut module: CUmodule = std::ptr::null_mut();
        result = cuda::cuModuleLoadData(&mut module, ptx_cstr.as_ptr() as *const _);
        check_error(result, "cuModuleLoadData")?;
        
        // Get function
        let mut function: CUfunction = std::ptr::null_mut();
        let func_name = CString::new("vector_add")?;
        result = cuda::cuModuleGetFunction(&mut function, module, func_name.as_ptr());
        check_error(result, "cuModuleGetFunction")?;
        
        // Allocate device memory
        let size = n * std::mem::size_of::<f32>();
        let mut d_a: cuda_sys::cuda::CUdeviceptr = 0;
        let mut d_b: cuda_sys::cuda::CUdeviceptr = 0;
        let mut d_c: cuda_sys::cuda::CUdeviceptr = 0;
        
        result = cuda::cuMemAlloc_v2(&mut d_a, size);
        check_error(result, "cuMemAlloc d_a")?;
        
        result = cuda::cuMemAlloc_v2(&mut d_b, size);
        check_error(result, "cuMemAlloc d_b")?;
        
        result = cuda::cuMemAlloc_v2(&mut d_c, size);
        check_error(result, "cuMemAlloc d_c")?;
        
        // Copy to device (H2D)
        result = cuda::cuMemcpyHtoD_v2(d_a, a.as_ptr() as *const _, size);
        check_error(result, "cuMemcpyHtoD a")?;
        
        result = cuda::cuMemcpyHtoD_v2(d_b, b.as_ptr() as *const _, size);
        check_error(result, "cuMemcpyHtoD b")?;
        
        // Launch kernel
        let block_size = 256;
        let grid_size = (n as u32 + block_size - 1) / block_size;
        
        let mut args: [*mut std::ffi::c_void; 4] = [
            &d_a as *const _ as *mut _,
            &d_b as *const _ as *mut _,
            &d_c as *const _ as *mut _,
            &n as *const _ as *mut _,
        ];
        
        result = cuda::cuLaunchKernel(
            function,
            grid_size, 1, 1,      // grid dim
            block_size, 1, 1,     // block dim
            0,                    // shared memory
            std::ptr::null_mut(), // stream
            args.as_mut_ptr() as *mut *mut std::ffi::c_void,
            std::ptr::null_mut(), // extra
        );
        check_error(result, "cuLaunchKernel")?;
        
        // Wait for completion
        result = cuda::cuCtxSynchronize();
        check_error(result, "cuCtxSynchronize")?;
        
        // Copy result back (D2H)
        let mut c = vec![0.0f32; n];
        result = cuda::cuMemcpyDtoH_v2(c.as_mut_ptr() as *mut _, d_c, size);
        check_error(result, "cuMemcpyDtoH")?;
        
        // Cleanup
        cuda::cuMemFree_v2(d_a);
        cuda::cuMemFree_v2(d_b);
        cuda::cuMemFree_v2(d_c);
        cuda::cuModuleUnload(module);
        cuda::cuCtxDestroy_v2(context);
        
        Ok(c)
    }
}

fn check_error(result: CUresult, context: &str) -> Result<(), String> {
    if result != cuda_sys::cuda::CUresult::CUDA_SUCCESS {
        Err(format!("{} failed: {:?}", context, result))
    } else {
        Ok(())
    }
}
