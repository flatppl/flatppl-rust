module {
  func.func @sample(%key: tensor<2xui64>) -> (tensor<3xf32>, tensor<2xui64>) {
    %1, %2 = stablehlo.rng_bit_generator %key, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<4xui32>)
    %3 = stablehlo.constant dense<9> : tensor<4xui32>
    %4 = stablehlo.shift_right_logical %2, %3 : tensor<4xui32>
    %5 = stablehlo.convert %4 : (tensor<4xui32>) -> tensor<4xf32>
    %6 = stablehlo.constant dense<1.1920929E-7> : tensor<4xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<4xf32>
    %11 = stablehlo.constant dense<0.20000000298023224> : tensor<f32>
    %14 = stablehlo.constant dense<0.5> : tensor<f32>
    %18 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %19 = stablehlo.constant dense<[0.0, 0.20000000298023224, 0.5]> : tensor<3xf32>
    %20 = stablehlo.reshape %11 : (tensor<f32>) -> tensor<1xf32>
    %21 = stablehlo.reshape %14 : (tensor<f32>) -> tensor<1xf32>
    %22 = stablehlo.reshape %18 : (tensor<f32>) -> tensor<1xf32>
    %23 = stablehlo.concatenate %20, %21, %22, dim = 0 : (tensor<1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<3xf32>
    %24 = stablehlo.constant dense<1.0> : tensor<3xf32>
    %25 = stablehlo.constant dense<0.0> : tensor<3xf32>
    %26 = stablehlo.constant dense<0> : tensor<i32>
    %29:2 = stablehlo.while(%27 = %26, %28 = %25) : tensor<i32>, tensor<3xf32>
    cond {
      %30 = stablehlo.constant dense<4> : tensor<i32>
      %31 = stablehlo.compare LT, %27, %30, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %31 : tensor<i1>
    } do {
      %32 = stablehlo.dynamic_slice %7, %27, sizes = [1] : (tensor<4xf32>, tensor<i32>) -> tensor<1xf32>
      %33 = stablehlo.reshape %32 : (tensor<1xf32>) -> tensor<f32>
      %34 = stablehlo.broadcast_in_dim %33, dims = [] : (tensor<f32>) -> tensor<3xf32>
      %35 = stablehlo.compare GE, %34, %19 : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xi1>
      %36 = stablehlo.compare LT, %34, %23 : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xi1>
      %37 = stablehlo.and %35, %36 : tensor<3xi1>
      %38 = stablehlo.select %37, %24, %25 : (tensor<3xi1>, tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
      %39 = stablehlo.add %28, %38 : tensor<3xf32>
      %40 = stablehlo.constant dense<1> : tensor<i32>
      %41 = stablehlo.add %27, %40 : tensor<i32>
      stablehlo.return %41, %39 : tensor<i32>, tensor<3xf32>
    }
    return %29#1, %1 : tensor<3xf32>, tensor<2xui64>
  }
}
