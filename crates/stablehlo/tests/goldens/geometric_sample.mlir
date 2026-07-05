module {
  func.func @sample() -> tensor<f32> {
    %0 = stablehlo.constant dense<0.3> : tensor<f32>
    %1 = stablehlo.constant dense<0.0> : tensor<f32>
    %2 = stablehlo.constant dense<1.0> : tensor<f32>
    %3 = stablehlo.constant dense<> : tensor<0xi64>
    %4 = stablehlo.rng %1, %2, %3, distribution = UNIFORM : (tensor<f32>, tensor<f32>, tensor<0xi64>) -> tensor<f32>
    %5 = stablehlo.log %4 : tensor<f32>
    %6 = stablehlo.subtract %2, %0 : tensor<f32>
    %7 = stablehlo.log %6 : tensor<f32>
    %8 = stablehlo.divide %5, %7 : tensor<f32>
    %9 = stablehlo.floor %8 : tensor<f32>
    return %9 : tensor<f32>
  }
}
