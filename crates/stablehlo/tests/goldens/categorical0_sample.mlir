module {
  func.func @sample() -> tensor<f32> {
    %0 = stablehlo.constant dense<0.2> : tensor<f32>
    %1 = stablehlo.constant dense<0.3> : tensor<f32>
    %2 = stablehlo.constant dense<0.5> : tensor<f32>
    %3 = stablehlo.reshape %0 : (tensor<f32>) -> tensor<1xf32>
    %4 = stablehlo.reshape %1 : (tensor<f32>) -> tensor<1xf32>
    %5 = stablehlo.reshape %2 : (tensor<f32>) -> tensor<1xf32>
    %6 = stablehlo.concatenate %3, %4, %5, dim = 0 : (tensor<1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<3xf32>
    %7 = stablehlo.constant dense<0.0> : tensor<f32>
    %8 = stablehlo.constant dense<1.0> : tensor<f32>
    %9 = stablehlo.constant dense<> : tensor<0xi64>
    %10 = stablehlo.rng %7, %8, %9, distribution = UNIFORM : (tensor<f32>, tensor<f32>, tensor<0xi64>) -> tensor<f32>
    %11 = stablehlo.constant dense<0.0> : tensor<f32>
    %12 = stablehlo.constant dense<0.0> : tensor<f32>
    %13 = stablehlo.slice %6 [0:1] : (tensor<3xf32>) -> tensor<1xf32>
    %14 = stablehlo.reshape %13 : (tensor<1xf32>) -> tensor<f32>
    %15 = stablehlo.add %11, %14 : tensor<f32>
    %16 = stablehlo.compare LT, %15, %10 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %17 = stablehlo.select %16, %8, %7 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %18 = stablehlo.add %12, %17 : tensor<f32>
    %19 = stablehlo.slice %6 [1:2] : (tensor<3xf32>) -> tensor<1xf32>
    %20 = stablehlo.reshape %19 : (tensor<1xf32>) -> tensor<f32>
    %21 = stablehlo.add %15, %20 : tensor<f32>
    %22 = stablehlo.compare LT, %21, %10 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %23 = stablehlo.select %22, %8, %7 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %24 = stablehlo.add %18, %23 : tensor<f32>
    return %24 : tensor<f32>
  }
}
