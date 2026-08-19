module {
  func.func @logdensity() -> tensor<3x2xf32> {
    %0 = stablehlo.constant dense<1.0> : tensor<f32>
    %1 = stablehlo.constant dense<2.0> : tensor<f32>
    %2 = stablehlo.constant dense<3.0> : tensor<f32>
    %3 = stablehlo.reshape %0 : (tensor<f32>) -> tensor<1xf32>
    %4 = stablehlo.reshape %1 : (tensor<f32>) -> tensor<1xf32>
    %5 = stablehlo.reshape %2 : (tensor<f32>) -> tensor<1xf32>
    %6 = stablehlo.concatenate %3, %4, %5, dim = 0 : (tensor<1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<3xf32>
    %7 = stablehlo.constant dense<4.0> : tensor<f32>
    %8 = stablehlo.constant dense<5.0> : tensor<f32>
    %9 = stablehlo.constant dense<6.0> : tensor<f32>
    %10 = stablehlo.reshape %7 : (tensor<f32>) -> tensor<1xf32>
    %11 = stablehlo.reshape %8 : (tensor<f32>) -> tensor<1xf32>
    %12 = stablehlo.reshape %9 : (tensor<f32>) -> tensor<1xf32>
    %13 = stablehlo.concatenate %10, %11, %12, dim = 0 : (tensor<1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<3xf32>
    %14 = stablehlo.reshape %6 : (tensor<3xf32>) -> tensor<1x3xf32>
    %15 = stablehlo.reshape %13 : (tensor<3xf32>) -> tensor<1x3xf32>
    %16 = stablehlo.concatenate %14, %15, dim = 0 : (tensor<1x3xf32>, tensor<1x3xf32>) -> tensor<2x3xf32>
    %17 = stablehlo.transpose %16, dims = [1, 0] : (tensor<2x3xf32>) -> tensor<3x2xf32>
    return %17 : tensor<3x2xf32>
  }
}
