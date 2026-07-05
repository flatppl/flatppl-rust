module {
  func.func @sample() -> tensor<f32> {
    %0 = stablehlo.constant dense<0.0> : tensor<f32>
    %1 = stablehlo.constant dense<1.0> : tensor<f32>
    %2 = stablehlo.constant dense<0.0> : tensor<f32>
    %3 = stablehlo.constant dense<1.0> : tensor<f32>
    %4 = stablehlo.constant dense<> : tensor<0xi64>
    %5 = stablehlo.rng %2, %3, %4, distribution = UNIFORM : (tensor<f32>, tensor<f32>, tensor<0xi64>) -> tensor<f32>
    %6 = stablehlo.subtract %3, %5 : tensor<f32>
    %7 = stablehlo.divide %5, %6 : tensor<f32>
    %8 = stablehlo.log %7 : tensor<f32>
    %9 = stablehlo.multiply %1, %8 : tensor<f32>
    %10 = stablehlo.add %0, %9 : tensor<f32>
    return %10 : tensor<f32>
  }
}
