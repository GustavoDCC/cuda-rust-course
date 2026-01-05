mod kernel;

fn main() {
    println!("=== CUDA Performance Suite ===\n");
    
    // Run small test first
    test_small_gpu();
    
    // Run performance benchmark
    performance_benchmark();
    
    println!("\n=== All tests completed ===");
}

fn test_small_gpu() {
    println!("--- Small GPU Test (10 elements) ---");
    
    let n = 10;
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let b = vec![10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
    
    println!("Input A: {:?}", a);
    println!("Input B: {:?}", b);
    
    // Check what function name your kernel module actually has
    // It should be either: vector_add_gpu or run_vector_add
    match kernel::run_vector_add(&a, &b) {
        Ok(c_gpu) => {
            println!("GPU Result: {:?}", c_gpu);
            
            // Verify against CPU
            let mut c_cpu = vec![0.0; n];
            for i in 0..n {
                c_cpu[i] = a[i] + b[i];
            }
            
            if c_cpu == c_gpu {
                println!("✅ GPU computation works correctly!\n");
            } else {
                println!("❌ GPU result doesn't match CPU\n");
            }
        }
        Err(e) => {
            // Try alternative function name
            println!("First attempt failed: {}", e);
            println!("Trying alternative function name...\n");
            
            // If you named it run_vector_add in kernel.rs
            match kernel::run_vector_add(&a, &b) {
                Ok(c_gpu) => {
                    println!("GPU Result: {:?}", c_gpu);
                    
                    let mut c_cpu = vec![0.0; n];
                    for i in 0..n {
                        c_cpu[i] = a[i] + b[i];
                    }
                    
                    if c_cpu == c_gpu {
                        println!("✅ GPU computation works correctly!\n");
                    } else {
                        println!("❌ GPU result doesn't match CPU\n");
                    }
                }
                Err(e2) => {
                    println!("❌ Both attempts failed: {} and {}", e, e2);
                    println!("Check your kernel.rs function name\n");
                }
            }
        }
    }
}

fn performance_benchmark() {
    println!("\n=== PERFORMANCE BENCHMARK ===");
    println!("Testing CPU vs GPU for different sizes\n");
    
    // Test sizes from small to large
    let sizes = [
        1_000,       // 4KB - should be CPU faster
        10_000,      // 40KB
        100_000,     // 400KB
        1_000_000,   // 4MB
        5_000_000,   // 20MB
        10_000_000,  // 40MB
    ];
    
    println!("Size       | CPU Time    | GPU Time    | Speedup | H2D MB/s | D2H MB/s");
    println!("-----------|-------------|-------------|---------|----------|----------");
    
    for &size in &sizes {
        benchmark_single_size(size);
    }
    
    println!("\n✅ Benchmark complete!");
}

fn benchmark_single_size(n: usize) {
    // Create test data
    let a: Vec<f32> = vec![1.5; n];  // Constant values
    let b: Vec<f32> = vec![2.5; n];
    
    // === CPU TIMING ===
    let cpu_start = std::time::Instant::now();
    let mut c_cpu = vec![0.0; n];
    for i in 0..n {
        c_cpu[i] = a[i] + b[i];
    }
    let cpu_time = cpu_start.elapsed();
    
    // === GPU TIMING ===
    // Try both function names
    let gpu_start = std::time::Instant::now();
    let result = kernel::run_vector_add(&a, &b)
        .or_else(|_| kernel::run_vector_add(&a, &b));
    let gpu_time = gpu_start.elapsed();
    
    match result {
        Ok(c_gpu) => {
            // Verify
            let mut correct = true;
            let expected = 4.0; // 1.5 + 2.5 = 4.0
            let check_count = n.min(100);
            for i in 0..check_count {
                if (c_gpu[i] - expected).abs() > 0.001 {
                    correct = false;
                    break;
                }
            }
            
            let speedup = if gpu_time.as_nanos() > 0 {
                cpu_time.as_secs_f64() / gpu_time.as_secs_f64()
            } else {
                0.0
            };
            
            let data_mb = (n * std::mem::size_of::<f32>()) as f64 / 1_000_000.0;
            
            println!("{:10} | {:7.2?} | {:7.2?} | {:6.1}x | {:8.1} | {:8.1}",
                n,
                cpu_time,
                gpu_time,
                speedup,
                data_mb / cpu_time.as_secs_f64(),
                data_mb / gpu_time.as_secs_f64()
            );
            
            if !correct {
                println!("   ⚠️  Data verification failed!");
            }
        }
        Err(e) => {
            println!("{:10} | {:7.2?} | ❌ Failed: {}", n, cpu_time, e);
        }
    }
}
