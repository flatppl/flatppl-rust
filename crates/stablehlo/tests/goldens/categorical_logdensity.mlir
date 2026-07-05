module {
  func.func @logdensity() -> tensor<f32> {
    %0 = stablehlo.constant dense<2.0> : tensor<f32>
    %1 = stablehlo.constant dense<0.2> : tensor<f32>
    %2 = stablehlo.constant dense<0.3> : tensor<f32>
    %3 = stablehlo.constant dense<0.5> : tensor<f32>
    %4 = stablehlo.reshape %1 : (tensor<f32>) -> tensor<1xf32>
    %5 = stablehlo.reshape %2 : (tensor<f32>) -> tensor<1xf32>
    %6 = stablehlo.reshape %3 : (tensor<f32>) -> tensor<1xf32>
    %7 = stablehlo.concatenate %4, %5, %6, dim = 0 : (tensor<1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<3xf32>
    %8 = stablehlo.slice %7 [1:2] : (tensor<3xf32>) -> tensor<1xf32>
    %9 = stablehlo.reshape %8 : (tensor<1xf32>) -> tensor<f32>
    %10 = stablehlo.log %9 : tensor<f32>
    return %10 : tensor<f32>
  }
}
