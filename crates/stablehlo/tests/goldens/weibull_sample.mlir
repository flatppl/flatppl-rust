module {
  func.func @sample() -> tensor<f32> {
    %0 = stablehlo.constant dense<2.0> : tensor<f32>
    %1 = stablehlo.constant dense<3.0> : tensor<f32>
    %2 = stablehlo.constant dense<0.0> : tensor<f32>
    %3 = stablehlo.constant dense<1.0> : tensor<f32>
    %4 = stablehlo.constant dense<> : tensor<0xi64>
    %5 = stablehlo.rng %2, %3, %4, distribution = UNIFORM : (tensor<f32>, tensor<f32>, tensor<0xi64>) -> tensor<f32>
    %6 = stablehlo.log %5 : tensor<f32>
    %7 = stablehlo.negate %6 : tensor<f32>
    %8 = stablehlo.divide %3, %0 : tensor<f32>
    %9 = stablehlo.power %7, %8 : tensor<f32>
    %10 = stablehlo.multiply %1, %9 : tensor<f32>
    return %10 : tensor<f32>
  }
}
