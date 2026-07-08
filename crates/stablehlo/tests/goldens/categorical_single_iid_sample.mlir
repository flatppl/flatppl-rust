module {
  func.func @sample(%key: tensor<2xui64>) -> (tensor<4xf32>, tensor<2xui64>) {
    %0 = stablehlo.constant dense<1.0> : tensor<f32>
    %1 = stablehlo.reshape %0 : (tensor<f32>) -> tensor<1xf32>
    %2 = stablehlo.concatenate %1, dim = 0 : (tensor<1xf32>) -> tensor<1xf32>
    %3 = stablehlo.constant dense<0.0> : tensor<f32>
    %4 = stablehlo.constant dense<1.0> : tensor<f32>
    %5, %6 = stablehlo.rng_bit_generator %key, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<4xui32>)
    %7 = stablehlo.constant dense<9> : tensor<4xui32>
    %8 = stablehlo.shift_right_logical %6, %7 : tensor<4xui32>
    %9 = stablehlo.convert %8 : (tensor<4xui32>) -> tensor<4xf32>
    %10 = stablehlo.constant dense<1.1920929E-7> : tensor<4xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<4xf32>
    %12 = stablehlo.constant dense<0.0> : tensor<f32>
    %13 = stablehlo.constant dense<1.0> : tensor<f32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [] : (tensor<f32>) -> tensor<4xf32>
    return %14, %5 : tensor<4xf32>, tensor<2xui64>
  }
}
