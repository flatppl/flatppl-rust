module {
  func.func @logdensity(%arg0: tensor<f32>) -> tensor<f32> {
    %1 = stablehlo.log %arg0 : tensor<f32>
    %2 = stablehlo.constant dense<"0x0000803F"> : tensor<f32>
    %3 = stablehlo.multiply %2, %1 : tensor<f32>
    %4 = stablehlo.constant dense<1.0> : tensor<f32>
    %5 = stablehlo.constant dense<"0x00000000"> : tensor<f32>
    %6 = stablehlo.subtract %4, %arg0 : tensor<f32>
    %7 = stablehlo.log %6 : tensor<f32>
    %8 = stablehlo.multiply %5, %7 : tensor<f32>
    %9 = stablehlo.add %3, %8 : tensor<f32>
    return %9 : tensor<f32>
  }
}
