module {
  func.func @sample(%key: tensor<2xui64>) -> (tensor<4xf32>, tensor<2xui64>) {
    %1 = stablehlo.constant dense<0.0> : tensor<f32>
    %2 = stablehlo.constant dense<1.0> : tensor<f32>
    %3, %4 = stablehlo.rng_bit_generator %key, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<4xui32>)
    %5 = stablehlo.constant dense<9> : tensor<4xui32>
    %6 = stablehlo.shift_right_logical %4, %5 : tensor<4xui32>
    %7 = stablehlo.convert %6 : (tensor<4xui32>) -> tensor<4xf32>
    %8 = stablehlo.constant dense<1.1920929E-7> : tensor<4xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<4xf32>
    %12 = stablehlo.constant dense<0.20000000298023224> : tensor<f32>
    %13 = stablehlo.broadcast_in_dim %12, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %14 = stablehlo.compare LT, %13, %9 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
    %15 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %16 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %17 = stablehlo.select %14, %15, %16 : (tensor<4xi1>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    %18 = stablehlo.add %15, %17 : tensor<4xf32>
    %21 = stablehlo.constant dense<0.5> : tensor<f32>
    %22 = stablehlo.broadcast_in_dim %21, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %23 = stablehlo.compare LT, %22, %9 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
    %24 = stablehlo.select %23, %15, %16 : (tensor<4xi1>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    %25 = stablehlo.add %18, %24 : tensor<4xf32>
    return %25, %3 : tensor<4xf32>, tensor<2xui64>
  }
}
