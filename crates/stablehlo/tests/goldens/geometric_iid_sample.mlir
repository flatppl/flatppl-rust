module {
  func.func @sample(%key: tensor<2xui64>) -> (tensor<4xf32>, tensor<2xui64>) {
    %0 = stablehlo.constant dense<0.3> : tensor<f32>
    %1 = stablehlo.constant dense<1.0> : tensor<f32>
    %2, %3 = stablehlo.rng_bit_generator %key, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<4xui32>)
    %4 = stablehlo.constant dense<9> : tensor<4xui32>
    %5 = stablehlo.shift_right_logical %3, %4 : tensor<4xui32>
    %6 = stablehlo.convert %5 : (tensor<4xui32>) -> tensor<4xf32>
    %7 = stablehlo.constant dense<1.1920929E-7> : tensor<4xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<4xf32>
    %9 = stablehlo.log %8 : tensor<4xf32>
    %10 = stablehlo.subtract %1, %0 : tensor<f32>
    %11 = stablehlo.log %10 : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %13 = stablehlo.divide %9, %12 : tensor<4xf32>
    %14 = stablehlo.floor %13 : tensor<4xf32>
    return %14, %2 : tensor<4xf32>, tensor<2xui64>
  }
}
