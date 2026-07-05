module {
  func.func @sample() -> tensor<f32> {
    %0 = stablehlo.constant dense<4.0> : tensor<f32>
    %1 = stablehlo.constant dense<0.0> : tensor<f32>
    %2 = stablehlo.constant dense<1.0> : tensor<f32>
    %3 = stablehlo.constant dense<> : tensor<0xi64>
    %4 = stablehlo.rng %1, %2, %3, distribution = UNIFORM : (tensor<f32>, tensor<f32>, tensor<0xi64>) -> tensor<f32>
    %5 = stablehlo.negate %0 : tensor<f32>
    %6 = stablehlo.exponential %5 : tensor<f32>
    %7 = stablehlo.constant dense<0.0> : tensor<f32>
    %8 = stablehlo.constant dense<false> : tensor<i1>
    %9 = stablehlo.constant dense<0.0> : tensor<f32>
    %15:5 = stablehlo.while(%10 = %7, %11 = %6, %12 = %6, %13 = %8, %14 = %9) : tensor<f32>, tensor<f32>, tensor<f32>, tensor<i1>, tensor<f32>
    cond {
      %16 = stablehlo.constant dense<256.0> : tensor<f32>
      %17 = stablehlo.compare LT, %10, %16 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %18 = stablehlo.not %13 : tensor<i1>
      %19 = stablehlo.and %18, %17 : tensor<i1>
      stablehlo.return %19 : tensor<i1>
    } do {
      %20 = stablehlo.compare LE, %4, %11 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %21 = stablehlo.constant dense<1.0> : tensor<f32>
      %22 = stablehlo.add %10, %21 : tensor<f32>
      %23 = stablehlo.divide %0, %22 : tensor<f32>
      %24 = stablehlo.multiply %12, %23 : tensor<f32>
      %25 = stablehlo.add %11, %24 : tensor<f32>
      stablehlo.return %22, %25, %24, %20, %10 : tensor<f32>, tensor<f32>, tensor<f32>, tensor<i1>, tensor<f32>
    }
    return %15#4 : tensor<f32>
  }
}
