module {
  func.func @sample() -> tensor<f32> {
    %0 = stablehlo.constant dense<0.0> : tensor<f32>
    %1 = stablehlo.constant dense<1.0> : tensor<f32>
    %2 = stablehlo.constant dense<> : tensor<0xi64>
    %3 = stablehlo.rng %0, %1, %2, distribution = UNIFORM : (tensor<f32>, tensor<f32>, tensor<0xi64>) -> tensor<f32>
    %4 = stablehlo.constant dense<-1.0> : tensor<f32>
    %5 = stablehlo.constant dense<4.0> : tensor<f32>
    %6 = stablehlo.multiply %5, %3 : tensor<f32>
    %7 = stablehlo.add %4, %6 : tensor<f32>
    return %7 : tensor<f32>
  }
}
