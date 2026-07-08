module {
  func.func @sample(%key: tensor<2xui64>) -> (tensor<4xf32>, tensor<2xui64>) {
    %0 = stablehlo.constant dense<2.0> : tensor<f32>
    %1, %2 = stablehlo.rng_bit_generator %key, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<4xui32>)
    %3 = stablehlo.constant dense<9> : tensor<4xui32>
    %4 = stablehlo.shift_right_logical %2, %3 : tensor<4xui32>
    %5 = stablehlo.convert %4 : (tensor<4xui32>) -> tensor<4xf32>
    %6 = stablehlo.constant dense<1.1920929E-7> : tensor<4xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<4xf32>
    %8 = stablehlo.log %7 : tensor<4xf32>
    %9 = stablehlo.negate %8 : tensor<4xf32>
    %10 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %11 = stablehlo.divide %9, %10 : tensor<4xf32>
    return %11, %1 : tensor<4xf32>, tensor<2xui64>
  }
}
