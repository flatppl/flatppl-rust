module {
  func.func @sample() -> tensor<f32> {
    %0 = stablehlo.constant dense<0.3> : tensor<f32>
    %1 = stablehlo.constant dense<0.0> : tensor<f32>
    %2 = stablehlo.constant dense<1.0> : tensor<f32>
    %3 = stablehlo.constant dense<5> : tensor<1xi64>
    %4 = stablehlo.rng %1, %2, %3, distribution = UNIFORM : (tensor<f32>, tensor<f32>, tensor<1xi64>) -> tensor<5xf32>
    %5 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<5xf32>
    %6 = stablehlo.compare LT, %4, %5 : (tensor<5xf32>, tensor<5xf32>) -> tensor<5xi1>
    %7 = stablehlo.constant dense<1.0> : tensor<5xf32>
    %8 = stablehlo.constant dense<0.0> : tensor<5xf32>
    %9 = stablehlo.select %6, %7, %8 : (tensor<5xi1>, tensor<5xf32>, tensor<5xf32>) -> tensor<5xf32>
    %10 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %11 = stablehlo.reduce(%9 init: %10) applies stablehlo.add across dimensions = [0] : (tensor<5xf32>, tensor<f32>) -> tensor<f32>
    return %11 : tensor<f32>
  }
}
