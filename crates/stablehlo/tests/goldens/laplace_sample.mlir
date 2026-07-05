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
    %8 = stablehlo.compare GE, %7, %2 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %9 = stablehlo.constant dense<1.0> : tensor<f32>
    %10 = stablehlo.constant dense<-1.0> : tensor<f32>
    %11 = stablehlo.select %8, %9, %10 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %12 = stablehlo.abs %7 : tensor<f32>
    %13 = stablehlo.constant dense<2.0> : tensor<f32>
    %14 = stablehlo.multiply %13, %12 : tensor<f32>
    %15 = stablehlo.subtract %3, %14 : tensor<f32>
    %16 = stablehlo.log %15 : tensor<f32>
    %17 = stablehlo.multiply %11, %16 : tensor<f32>
    %18 = stablehlo.multiply %1, %17 : tensor<f32>
    %19 = stablehlo.negate %18 : tensor<f32>
    %20 = stablehlo.add %0, %19 : tensor<f32>
    return %20 : tensor<f32>
  }
}
