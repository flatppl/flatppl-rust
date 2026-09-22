module {
  func.func @sample(%key: tensor<2xui64>) -> (tensor<f32>, tensor<2xui64>) {
    %0 = stablehlo.constant dense<4.0> : tensor<f32>
    %1, %2 = stablehlo.rng_bit_generator %key, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<ui32>)
    %3 = stablehlo.constant dense<9> : tensor<ui32>
    %4 = stablehlo.shift_right_logical %2, %3 : tensor<ui32>
    %5 = stablehlo.convert %4 : (tensor<ui32>) -> tensor<f32>
    %6 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
    %7 = stablehlo.multiply %5, %6 : tensor<f32>
    %9 = stablehlo.constant dense<0.018315639346837997> : tensor<f32>
    %10 = stablehlo.constant dense<0.0> : tensor<f32>
    %11 = stablehlo.constant dense<false> : tensor<i1>
    %17:5 = stablehlo.while(%12 = %10, %13 = %9, %14 = %9, %15 = %11, %16 = %10) : tensor<f32>, tensor<f32>, tensor<f32>, tensor<i1>, tensor<f32>
    cond {
      %18 = stablehlo.constant dense<256.0> : tensor<f32>
      %19 = stablehlo.compare LT, %12, %18 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %20 = stablehlo.not %15 : tensor<i1>
      %21 = stablehlo.and %20, %19 : tensor<i1>
      stablehlo.return %21 : tensor<i1>
    } do {
      %22 = stablehlo.compare LE, %7, %13 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %23 = stablehlo.constant dense<1.0> : tensor<f32>
      %24 = stablehlo.add %12, %23 : tensor<f32>
      %25 = stablehlo.divide %0, %24 : tensor<f32>
      %26 = stablehlo.multiply %14, %25 : tensor<f32>
      %27 = stablehlo.add %13, %26 : tensor<f32>
      stablehlo.return %24, %27, %26, %22, %12 : tensor<f32>, tensor<f32>, tensor<f32>, tensor<i1>, tensor<f32>
    }
    return %17#4, %1 : tensor<f32>, tensor<2xui64>
  }
}
