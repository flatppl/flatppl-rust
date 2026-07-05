module {
  func.func @sample() -> tensor<f32> {
    %0 = stablehlo.constant dense<0.0> : tensor<f32>
    %1 = stablehlo.constant dense<1.0> : tensor<f32>
    %2 = stablehlo.constant dense<0.0> : tensor<f32>
    %3 = stablehlo.constant dense<1.0> : tensor<f32>
    %4 = stablehlo.constant dense<> : tensor<0xi64>
    %5 = stablehlo.rng %2, %3, %4, distribution = UNIFORM : (tensor<f32>, tensor<f32>, tensor<0xi64>) -> tensor<f32>
    %6 = stablehlo.constant dense<0.5> : tensor<f32>
    %7 = stablehlo.subtract %5, %6 : tensor<f32>
    %8 = stablehlo.constant dense<3.141592653589793> : tensor<f32>
    %9 = stablehlo.multiply %8, %7 : tensor<f32>
    %10 = stablehlo.sine %9 : tensor<f32>
    %11 = stablehlo.cosine %9 : tensor<f32>
    %12 = stablehlo.divide %10, %11 : tensor<f32>
    %13 = stablehlo.multiply %1, %12 : tensor<f32>
    %14 = stablehlo.add %0, %13 : tensor<f32>
    return %14 : tensor<f32>
  }
}
