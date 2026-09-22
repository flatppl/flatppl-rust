module {
  func.func @logdensity(%arg0: tensor<f32>) -> tensor<f32> {
    %1 = stablehlo.log %arg0 : tensor<f32>
    %2 = stablehlo.constant dense<3.0> : tensor<f32>
    %3 = stablehlo.multiply %2, %1 : tensor<f32>
    %4 = stablehlo.negate %arg0 : tensor<f32>
    %8 = stablehlo.constant dense<-1.7917594909667969> : tensor<f32>
    %9 = stablehlo.add %3, %4 : tensor<f32>
    %10 = stablehlo.add %9, %8 : tensor<f32>
    return %10 : tensor<f32>
  }
}
