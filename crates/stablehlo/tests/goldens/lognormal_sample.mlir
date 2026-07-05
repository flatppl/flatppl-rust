module {
  func.func @sample() -> tensor<f32> {
    %0 = stablehlo.constant dense<0.0> : tensor<f32>
    %1 = stablehlo.constant dense<1.0> : tensor<f32>
    %2 = stablehlo.constant dense<0.0> : tensor<f32>
    %3 = stablehlo.constant dense<1.0> : tensor<f32>
    %4 = stablehlo.constant dense<> : tensor<0xi64>
    %5 = stablehlo.rng %2, %3, %4, distribution = NORMAL : (tensor<f32>, tensor<f32>, tensor<0xi64>) -> tensor<f32>
    %6 = stablehlo.multiply %1, %5 : tensor<f32>
    %7 = stablehlo.add %0, %6 : tensor<f32>
    %8 = stablehlo.exponential %7 : tensor<f32>
    return %8 : tensor<f32>
  }
}
