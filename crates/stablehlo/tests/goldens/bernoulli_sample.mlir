module {
  func.func @sample() -> tensor<f32> {
    %0 = stablehlo.constant dense<0.3> : tensor<f32>
    %1 = stablehlo.constant dense<0.0> : tensor<f32>
    %2 = stablehlo.constant dense<1.0> : tensor<f32>
    %3 = stablehlo.constant dense<> : tensor<0xi64>
    %4 = stablehlo.rng %1, %2, %3, distribution = UNIFORM : (tensor<f32>, tensor<f32>, tensor<0xi64>) -> tensor<f32>
    %5 = stablehlo.compare LT, %4, %0 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %6 = stablehlo.select %5, %2, %1 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    return %6 : tensor<f32>
  }
}
