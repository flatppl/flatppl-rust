module {
  func.func @sample(%key: tensor<2xui64>) -> (tensor<4xf32>, tensor<2xui64>) {
    %0 = stablehlo.constant dense<0.0> : tensor<f32>
    %1 = stablehlo.constant dense<1.0> : tensor<f32>
    %2, %3 = stablehlo.rng_bit_generator %key, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<4xui32>)
    %4 = stablehlo.constant dense<9> : tensor<4xui32>
    %5 = stablehlo.shift_right_logical %3, %4 : tensor<4xui32>
    %6 = stablehlo.convert %5 : (tensor<4xui32>) -> tensor<4xf32>
    %7 = stablehlo.constant dense<1.1920929E-7> : tensor<4xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<4xf32>
    %9 = stablehlo.constant dense<2.0> : tensor<4xf32>
    %10 = stablehlo.constant dense<1.0> : tensor<4xf32>
    %11 = stablehlo.multiply %8, %9 : tensor<4xf32>
    %12 = stablehlo.subtract %11, %10 : tensor<4xf32>
    %13 = chlo.erf_inv %12 : tensor<4xf32> -> tensor<4xf32>
    %14 = stablehlo.constant dense<1.4142135> : tensor<4xf32>
    %15 = stablehlo.multiply %13, %14 : tensor<4xf32>
    %16 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %17 = stablehlo.multiply %16, %15 : tensor<4xf32>
    %18 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %19 = stablehlo.add %18, %17 : tensor<4xf32>
    return %19, %2 : tensor<4xf32>, tensor<2xui64>
  }
}
