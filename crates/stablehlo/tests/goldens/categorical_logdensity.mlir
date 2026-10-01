module {
  func.func @logdensity() -> tensor<f32> {
    %4 = stablehlo.constant dense<"0xC81B9ABF"> : tensor<f32>
    return %4 : tensor<f32>
  }
}
