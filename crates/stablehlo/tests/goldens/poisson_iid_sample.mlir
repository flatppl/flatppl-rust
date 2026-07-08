module {
  func.func @sample(%key: tensor<2xui64>) -> (tensor<4xf32>, tensor<2xui64>) {
    %0 = stablehlo.constant dense<3.0> : tensor<f32>
    %1, %2 = stablehlo.rng_bit_generator %key, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<4xui32>)
    %3 = stablehlo.constant dense<9> : tensor<4xui32>
    %4 = stablehlo.shift_right_logical %2, %3 : tensor<4xui32>
    %5 = stablehlo.convert %4 : (tensor<4xui32>) -> tensor<4xf32>
    %6 = stablehlo.constant dense<1.1920929E-7> : tensor<4xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<4xf32>
    %8 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %9 = stablehlo.negate %8 : tensor<4xf32>
    %10 = stablehlo.exponential %9 : tensor<4xf32>
    %11 = stablehlo.constant dense<0.0> : tensor<f32>
    %12 = stablehlo.constant dense<false> : tensor<4xi1>
    %13 = stablehlo.constant dense<0.0> : tensor<4xf32>
    %19:5 = stablehlo.while(%14 = %11, %15 = %10, %16 = %10, %17 = %12, %18 = %13) : tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<4xi1>, tensor<4xf32>
    cond {
      %20 = stablehlo.constant dense<256.0> : tensor<f32>
      %21 = stablehlo.compare LT, %14, %20 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %22 = stablehlo.constant dense<true> : tensor<i1>
      %23 = stablehlo.reduce(%17 init: %22) applies stablehlo.and across dimensions = [0] : (tensor<4xi1>, tensor<i1>) -> tensor<i1>
      %24 = stablehlo.not %23 : tensor<i1>
      %25 = stablehlo.and %21, %24 : tensor<i1>
      stablehlo.return %25 : tensor<i1>
    } do {
      %26 = stablehlo.compare LE, %7, %15 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
      %27 = stablehlo.broadcast_in_dim %14, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %28 = stablehlo.select %17, %18, %27 : (tensor<4xi1>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
      %29 = stablehlo.or %17, %26 : tensor<4xi1>
      %30 = stablehlo.constant dense<1.0> : tensor<f32>
      %31 = stablehlo.add %14, %30 : tensor<f32>
      %32 = stablehlo.broadcast_in_dim %31, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %33 = stablehlo.divide %8, %32 : tensor<4xf32>
      %34 = stablehlo.multiply %16, %33 : tensor<4xf32>
      %35 = stablehlo.add %15, %34 : tensor<4xf32>
      stablehlo.return %31, %35, %34, %29, %28 : tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<4xi1>, tensor<4xf32>
    }
    return %19#4, %1 : tensor<4xf32>, tensor<2xui64>
  }
}
