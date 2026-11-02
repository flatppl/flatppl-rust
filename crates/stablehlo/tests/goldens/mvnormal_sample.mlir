module attributes {flatppl.runtime_status = 2 : i64} {
  func.func @sample(%key: tensor<2xui64>, %arg0: tensor<2xf32>, %arg1: tensor<2x2xf32>) -> (tensor<2xf32>, tensor<2xui64>, tensor<i32>) {
    %0 = stablehlo.cholesky %arg1, lower = true : tensor<2x2xf32>
    %1 = stablehlo.iota dim = 0 : tensor<2x2xf32>
    %2 = stablehlo.iota dim = 1 : tensor<2x2xf32>
    %3 = stablehlo.compare EQ, %1, %2 : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xi1>
    %4 = stablehlo.constant dense<0.0> : tensor<2x2xf32>
    %5 = stablehlo.select %3, %0, %4 : (tensor<2x2xi1>, tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    %6 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %7 = stablehlo.reduce(%5 init: %6) applies stablehlo.add across dimensions = [1] : (tensor<2x2xf32>, tensor<f32>) -> tensor<2xf32>
    %8 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %9 = stablehlo.reduce(%7 init: %8) applies stablehlo.minimum across dimensions = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
    %10 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %11 = stablehlo.reduce(%7 init: %10) applies stablehlo.maximum across dimensions = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
    %12 = stablehlo.constant dense<0.0> : tensor<f32>
    %13 = stablehlo.compare GT, %9, %12 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %14 = stablehlo.compare LT, %11, %8 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %15 = stablehlo.select %13, %14, %13 : (tensor<i1>, tensor<i1>, tensor<i1>) -> tensor<i1>
    %16 = stablehlo.constant dense<0> : tensor<i32>
    %17 = stablehlo.constant dense<9> : tensor<i32>
    %18 = stablehlo.select %15, %16, %17 : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %19, %20 = stablehlo.rng_bit_generator %key, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<2xui32>)
    %21 = stablehlo.constant dense<9> : tensor<2xui32>
    %22 = stablehlo.shift_right_logical %20, %21 : tensor<2xui32>
    %23 = stablehlo.convert %22 : (tensor<2xui32>) -> tensor<2xf32>
    %24 = stablehlo.constant dense<1.1920929E-7> : tensor<2xf32>
    %25 = stablehlo.multiply %23, %24 : tensor<2xf32>
    %26 = stablehlo.constant dense<2.0> : tensor<2xf32>
    %27 = stablehlo.constant dense<1.0> : tensor<2xf32>
    %28 = stablehlo.multiply %25, %26 : tensor<2xf32>
    %29 = stablehlo.subtract %28, %27 : tensor<2xf32>
    %30 = chlo.erf_inv %29 : tensor<2xf32> -> tensor<2xf32>
    %31 = stablehlo.constant dense<1.4142135> : tensor<2xf32>
    %32 = stablehlo.multiply %30, %31 : tensor<2xf32>
    %33 = stablehlo.dot_general %0, %32, contracting_dims = [1] x [0], precision = [HIGHEST, HIGHEST] : (tensor<2x2xf32>, tensor<2xf32>) -> tensor<2xf32>
    %34 = stablehlo.add %arg0, %33 : tensor<2xf32>
    return %34, %19, %18 : tensor<2xf32>, tensor<2xui64>, tensor<i32>
  }
}
