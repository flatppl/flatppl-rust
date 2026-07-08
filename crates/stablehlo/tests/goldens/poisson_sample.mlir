module {
  func.func @sample(%key: tensor<2xui64>) -> (tensor<f32>, tensor<2xui64>) {
    %0 = stablehlo.constant dense<4.0> : tensor<f32>
    %1, %2 = stablehlo.rng_bit_generator %key, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<ui32>)
    %3 = stablehlo.constant dense<9> : tensor<ui32>
    %4 = stablehlo.shift_right_logical %2, %3 : tensor<ui32>
    %5 = stablehlo.convert %4 : (tensor<ui32>) -> tensor<f32>
    %6 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
    %7 = stablehlo.multiply %5, %6 : tensor<f32>
    %8 = stablehlo.negate %0 : tensor<f32>
    %9 = stablehlo.exponential %8 : tensor<f32>
    %10 = stablehlo.constant dense<0.0> : tensor<f32>
    %11 = stablehlo.constant dense<false> : tensor<i1>
    %12 = stablehlo.constant dense<0.0> : tensor<f32>
    %18:5 = stablehlo.while(%13 = %10, %14 = %9, %15 = %9, %16 = %11, %17 = %12) : tensor<f32>, tensor<f32>, tensor<f32>, tensor<i1>, tensor<f32>
    cond {
      %19 = stablehlo.constant dense<256.0> : tensor<f32>
      %20 = stablehlo.compare LT, %13, %19 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %21 = stablehlo.not %16 : tensor<i1>
      %22 = stablehlo.and %21, %20 : tensor<i1>
      stablehlo.return %22 : tensor<i1>
    } do {
      %23 = stablehlo.compare LE, %7, %14 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %24 = stablehlo.constant dense<1.0> : tensor<f32>
      %25 = stablehlo.add %13, %24 : tensor<f32>
      %26 = stablehlo.divide %0, %25 : tensor<f32>
      %27 = stablehlo.multiply %15, %26 : tensor<f32>
      %28 = stablehlo.add %14, %27 : tensor<f32>
      stablehlo.return %25, %28, %27, %23, %13 : tensor<f32>, tensor<f32>, tensor<f32>, tensor<i1>, tensor<f32>
    }
    return %18#4, %1 : tensor<f32>, tensor<2xui64>
  }
}
