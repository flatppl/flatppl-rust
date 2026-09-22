module {
  func.func @sample(%key: tensor<2xui64>) -> (tensor<4xf32>, tensor<2xui64>) {
    %2 = stablehlo.constant dense<1.0> : tensor<f32>
    %3, %4 = stablehlo.rng_bit_generator %key, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<4xui32>)
    %5 = stablehlo.constant dense<9> : tensor<4xui32>
    %6 = stablehlo.shift_right_logical %4, %5 : tensor<4xui32>
    %7 = stablehlo.convert %6 : (tensor<4xui32>) -> tensor<4xf32>
    %8 = stablehlo.constant dense<1.1920929E-7> : tensor<4xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<4xf32>
    %10 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f32>) -> tensor<4xf32>
    return %10, %3 : tensor<4xf32>, tensor<2xui64>
  }
}
