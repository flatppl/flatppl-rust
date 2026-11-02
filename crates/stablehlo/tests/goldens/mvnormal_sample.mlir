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
    %8 = stablehlo.constant dense<0.0> : tensor<f32>
    %9 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<f32>) -> tensor<2xf32>
    %11 = stablehlo.compare GT, %7, %10 : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
    %12 = stablehlo.broadcast_in_dim %9, dims = [] : (tensor<f32>) -> tensor<2xf32>
    %13 = stablehlo.compare LT, %7, %12 : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
    %14 = stablehlo.select %11, %13, %11 : (tensor<2xi1>, tensor<2xi1>, tensor<2xi1>) -> tensor<2xi1>
    %15 = stablehlo.constant dense<true> : tensor<i1>
    %16 = stablehlo.reduce(%14 init: %15) applies stablehlo.and across dimensions = [0] : (tensor<2xi1>, tensor<i1>) -> tensor<i1>
    %17 = stablehlo.constant dense<0> : tensor<i32>
    %18 = stablehlo.constant dense<9> : tensor<i32>
    %19 = stablehlo.select %16, %17, %18 : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %20, %21 = stablehlo.rng_bit_generator %key, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<2xui32>)
    %22 = stablehlo.constant dense<9> : tensor<2xui32>
    %23 = stablehlo.shift_right_logical %21, %22 : tensor<2xui32>
    %24 = stablehlo.convert %23 : (tensor<2xui32>) -> tensor<2xf32>
    %25 = stablehlo.constant dense<1.1920929E-7> : tensor<2xf32>
    %26 = stablehlo.multiply %24, %25 : tensor<2xf32>
    %27 = stablehlo.constant dense<2.0> : tensor<2xf32>
    %28 = stablehlo.constant dense<1.0> : tensor<2xf32>
    %29 = stablehlo.multiply %26, %27 : tensor<2xf32>
    %30 = stablehlo.subtract %29, %28 : tensor<2xf32>
    %31 = chlo.erf_inv %30 : tensor<2xf32> -> tensor<2xf32>
    %32 = stablehlo.constant dense<1.4142135> : tensor<2xf32>
    %33 = stablehlo.multiply %31, %32 : tensor<2xf32>
    %34 = stablehlo.dot_general %0, %33, contracting_dims = [1] x [0], precision = [HIGHEST, HIGHEST] : (tensor<2x2xf32>, tensor<2xf32>) -> tensor<2xf32>
    %35 = stablehlo.add %arg0, %34 : tensor<2xf32>
    return %35, %20, %19 : tensor<2xf32>, tensor<2xui64>, tensor<i32>
  }
}
