module {
  func.func @sample() -> tensor<f32> {
    %0 = stablehlo.constant dense<1.0> : tensor<f32>
    %1 = stablehlo.constant dense<0.0> : tensor<f32>
    %2 = stablehlo.constant dense<1.0> : tensor<f32>
    %3 = stablehlo.constant dense<> : tensor<0xi64>
    %4 = stablehlo.rng %1, %2, %3, distribution = UNIFORM : (tensor<f32>, tensor<f32>, tensor<0xi64>) -> tensor<f32>
    %5 = stablehlo.log %4 : tensor<f32>
    %6 = stablehlo.negate %5 : tensor<f32>
    %7 = stablehlo.divide %6, %0 : tensor<f32>
    return %7 : tensor<f32>
  }
}
