module {
  func.func @logdensity() -> tensor<f32> {
    %3 = stablehlo.constant dense<true> : tensor<i1>
    %4 = stablehlo.and %3, %3 : tensor<i1>
    %5 = stablehlo.constant dense<-1.3862943611198906> : tensor<f32>
    %6 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %7 = stablehlo.negate %6 : tensor<f32>
    %8 = stablehlo.select %4, %5, %7 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    return %8 : tensor<f32>
  }
}
