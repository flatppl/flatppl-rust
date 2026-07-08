module {
  func.func @sample(%key: tensor<2xui64>) -> (tensor<4xf32>, tensor<2xui64>) {
    %0 = stablehlo.constant dense<2.0> : tensor<f32>
    %1 = stablehlo.constant dense<0.0> : tensor<f32>
    %2 = stablehlo.constant dense<1.0> : tensor<f32>
    %3, %4 = stablehlo.rng_bit_generator %key, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<4xui32>)
    %5 = stablehlo.constant dense<9> : tensor<4xui32>
    %6 = stablehlo.shift_right_logical %4, %5 : tensor<4xui32>
    %7 = stablehlo.convert %6 : (tensor<4xui32>) -> tensor<4xf32>
    %8 = stablehlo.constant dense<1.1920929E-7> : tensor<4xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<4xf32>
    %10 = stablehlo.subtract %2, %1 : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %12 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %13 = stablehlo.multiply %9, %11 : tensor<4xf32>
    %14 = stablehlo.add %13, %12 : tensor<4xf32>
    %15 = stablehlo.log %14 : tensor<4xf32>
    %16 = stablehlo.negate %15 : tensor<4xf32>
    %17 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %18 = stablehlo.divide %16, %17 : tensor<4xf32>
    return %18, %3 : tensor<4xf32>, tensor<2xui64>
  }
}
