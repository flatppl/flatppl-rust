module {
  func.func @logdensity() -> tensor<f32> {
    %4 = stablehlo.constant dense<-1.2039728164672852> : tensor<f32>
    return %4 : tensor<f32>
  }
}
