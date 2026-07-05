module {
  func.func @logdensity(%arg0: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.constant dense<3.0> : tensor<f32>
    %1 = stablehlo.log %arg0 : tensor<f32>
    %2 = stablehlo.multiply %0, %1 : tensor<f32>
    %3 = stablehlo.negate %arg0 : tensor<f32>
    %4 = stablehlo.constant dense<1.0> : tensor<f32>
    %5 = stablehlo.add %0, %4 : tensor<f32>
    %6 = chlo.lgamma %5 : tensor<f32> -> tensor<f32>
    %7 = stablehlo.negate %6 : tensor<f32>
    %8 = stablehlo.add %2, %3 : tensor<f32>
    %9 = stablehlo.add %8, %7 : tensor<f32>
    return %9 : tensor<f32>
  }
}
