module {
  func.func @logdensity() -> tensor<f32> {
    %0 = stablehlo.constant dense<0.5> : tensor<f32>
    %1 = stablehlo.constant dense<-1.0> : tensor<f32>
    %2 = stablehlo.constant dense<3.0> : tensor<f32>
    %3 = stablehlo.compare GE, %0, %1 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %4 = stablehlo.compare LE, %0, %2 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %5 = stablehlo.and %3, %4 : tensor<i1>
    %6 = stablehlo.constant dense<-1.3862943611198906> : tensor<f32>
    %7 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %8 = stablehlo.negate %7 : tensor<f32>
    %9 = stablehlo.select %5, %6, %8 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    return %9 : tensor<f32>
  }
}
