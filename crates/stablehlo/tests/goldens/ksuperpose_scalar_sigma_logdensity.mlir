module {
  func.func @logdensity() -> tensor<f32> {
    %0 = stablehlo.constant dense<[0.3, 1.2]> : tensor<2xf32>
    %20 = stablehlo.constant dense<[-3.2479114532470703, -1.861617088317871]> : tensor<2xf32>
    %21 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %22 = stablehlo.reduce(%20 init: %21) applies stablehlo.maximum across dimensions = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
    %23 = stablehlo.broadcast_in_dim %22, dims = [] : (tensor<f32>) -> tensor<2xf32>
    %24 = stablehlo.subtract %20, %23 : tensor<2xf32>
    %25 = stablehlo.exponential %24 : tensor<2xf32>
    %26 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %27 = stablehlo.reduce(%25 init: %26) applies stablehlo.add across dimensions = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
    %28 = stablehlo.log %27 : tensor<f32>
    %29 = stablehlo.add %28, %22 : tensor<f32>
    %30 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %31 = stablehlo.reduce(%0 init: %30) applies stablehlo.add across dimensions = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
    %32 = stablehlo.log %31 : tensor<f32>
    %33 = stablehlo.subtract %29, %32 : tensor<f32>
    return %33 : tensor<f32>
  }
}
