module {
  func.func @logdensity() -> tensor<f32> {
    %0 = stablehlo.constant dense<0.5> : tensor<f32>
    %1 = stablehlo.constant dense<-1.3862943611198906> : tensor<f32>
    return %1 : tensor<f32>
  }
}
