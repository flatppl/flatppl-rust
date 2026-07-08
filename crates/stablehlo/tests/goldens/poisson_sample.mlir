module {
  func.func @sample(%key: tensor<2xui64>) -> (tensor<f32>, tensor<2xui64>) {
    %0 = stablehlo.constant dense<4.0> : tensor<f32>
    %1 = stablehlo.constant dense<0.0> : tensor<f32>
    %2 = stablehlo.constant dense<1.0> : tensor<f32>
    %3, %4 = stablehlo.rng_bit_generator %key, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<ui32>)
    %5 = stablehlo.constant dense<9> : tensor<ui32>
    %6 = stablehlo.shift_right_logical %4, %5 : tensor<ui32>
    %7 = stablehlo.convert %6 : (tensor<ui32>) -> tensor<f32>
    %8 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
    %9 = stablehlo.multiply %7, %8 : tensor<f32>
    %10 = stablehlo.subtract %2, %1 : tensor<f32>
    %11 = stablehlo.multiply %9, %10 : tensor<f32>
    %12 = stablehlo.add %11, %1 : tensor<f32>
    %13 = stablehlo.negate %0 : tensor<f32>
    %14 = stablehlo.exponential %13 : tensor<f32>
    %15 = stablehlo.constant dense<0.0> : tensor<f32>
    %16 = stablehlo.constant dense<false> : tensor<i1>
    %17 = stablehlo.constant dense<0.0> : tensor<f32>
    %23:5 = stablehlo.while(%18 = %15, %19 = %14, %20 = %14, %21 = %16, %22 = %17) : tensor<f32>, tensor<f32>, tensor<f32>, tensor<i1>, tensor<f32>
    cond {
      %24 = stablehlo.constant dense<256.0> : tensor<f32>
      %25 = stablehlo.compare LT, %18, %24 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %26 = stablehlo.not %21 : tensor<i1>
      %27 = stablehlo.and %26, %25 : tensor<i1>
      stablehlo.return %27 : tensor<i1>
    } do {
      %28 = stablehlo.compare LE, %12, %19 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %29 = stablehlo.constant dense<1.0> : tensor<f32>
      %30 = stablehlo.add %18, %29 : tensor<f32>
      %31 = stablehlo.divide %0, %30 : tensor<f32>
      %32 = stablehlo.multiply %20, %31 : tensor<f32>
      %33 = stablehlo.add %19, %32 : tensor<f32>
      stablehlo.return %30, %33, %32, %28, %18 : tensor<f32>, tensor<f32>, tensor<f32>, tensor<i1>, tensor<f32>
    }
    return %23#4, %3 : tensor<f32>, tensor<2xui64>
  }
}
