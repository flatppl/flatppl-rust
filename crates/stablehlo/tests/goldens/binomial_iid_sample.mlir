module {
  func.func @sample(%key: tensor<2xui64>) -> (tensor<4xf32>, tensor<2xui64>) {
    %0 = stablehlo.constant dense<0.3> : tensor<f32>
    %1, %2 = stablehlo.rng_bit_generator %key, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<4x5xui32>)
    %3 = stablehlo.constant dense<9> : tensor<4x5xui32>
    %4 = stablehlo.shift_right_logical %2, %3 : tensor<4x5xui32>
    %5 = stablehlo.convert %4 : (tensor<4x5xui32>) -> tensor<4x5xf32>
    %6 = stablehlo.constant dense<1.1920929E-7> : tensor<4x5xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<4x5xf32>
    %8 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<4x5xf32>
    %9 = stablehlo.compare LT, %7, %8 : (tensor<4x5xf32>, tensor<4x5xf32>) -> tensor<4x5xi1>
    %10 = stablehlo.constant dense<1.0> : tensor<4x5xf32>
    %11 = stablehlo.constant dense<0.0> : tensor<4x5xf32>
    %12 = stablehlo.select %9, %10, %11 : (tensor<4x5xi1>, tensor<4x5xf32>, tensor<4x5xf32>) -> tensor<4x5xf32>
    %13 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %14 = stablehlo.reduce(%12 init: %13) applies stablehlo.add across dimensions = [1] : (tensor<4x5xf32>, tensor<f32>) -> tensor<4xf32>
    return %14, %1 : tensor<4xf32>, tensor<2xui64>
  }
}
