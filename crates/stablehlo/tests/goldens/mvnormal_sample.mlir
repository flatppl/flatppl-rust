module {
  func.func @sample(%key: tensor<2xui64>, %arg0: tensor<2xf32>, %arg1: tensor<2x2xf32>) -> (tensor<2xf32>, tensor<2xui64>) {
    %0 = stablehlo.cholesky %arg1, lower = true : tensor<2x2xf32>
    %1 = stablehlo.constant dense<0.0> : tensor<f32>
    %2 = stablehlo.constant dense<1.0> : tensor<f32>
    %3, %4 = stablehlo.rng_bit_generator %key, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<2xui32>)
    %5 = stablehlo.constant dense<9> : tensor<2xui32>
    %6 = stablehlo.shift_right_logical %4, %5 : tensor<2xui32>
    %7 = stablehlo.convert %6 : (tensor<2xui32>) -> tensor<2xf32>
    %8 = stablehlo.constant dense<1.1920929E-7> : tensor<2xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<2xf32>
    %10 = stablehlo.constant dense<2.0> : tensor<2xf32>
    %11 = stablehlo.constant dense<1.0> : tensor<2xf32>
    %12 = stablehlo.multiply %9, %10 : tensor<2xf32>
    %13 = stablehlo.subtract %12, %11 : tensor<2xf32>
    %14 = chlo.erf_inv %13 : tensor<2xf32> -> tensor<2xf32>
    %15 = stablehlo.constant dense<1.4142135> : tensor<2xf32>
    %16 = stablehlo.multiply %14, %15 : tensor<2xf32>
    %17 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f32>) -> tensor<2xf32>
    %18 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<f32>) -> tensor<2xf32>
    %19 = stablehlo.multiply %16, %17 : tensor<2xf32>
    %20 = stablehlo.add %19, %18 : tensor<2xf32>
    %21 = stablehlo.dot_general %0, %20, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x2xf32>, tensor<2xf32>) -> tensor<2xf32>
    %22 = stablehlo.add %arg0, %21 : tensor<2xf32>
    return %22, %3 : tensor<2xf32>, tensor<2xui64>
  }
}
