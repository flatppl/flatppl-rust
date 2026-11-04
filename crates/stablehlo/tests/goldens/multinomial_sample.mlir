module {
  func.func @sample(%key: tensor<2xui64>) -> (tensor<3xi32>, tensor<2xui64>) {
    %2, %3 = stablehlo.rng_bit_generator %key, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<4xui32>)
    %4 = stablehlo.constant dense<9> : tensor<4xui32>
    %5 = stablehlo.shift_right_logical %3, %4 : tensor<4xui32>
    %6 = stablehlo.convert %5 : (tensor<4xui32>) -> tensor<4xf32>
    %7 = stablehlo.constant dense<1.1920929E-7> : tensor<4xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<4xf32>
    %12 = stablehlo.constant dense<"0xCDCC4C3E"> : tensor<f32>
    %15 = stablehlo.constant dense<"0x0000003F"> : tensor<f32>
    %19 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %20 = stablehlo.constant dense<"0x00000000CDCC4C3E0000003F"> : tensor<3xf32>
    %21 = stablehlo.reshape %12 : (tensor<f32>) -> tensor<1xf32>
    %22 = stablehlo.reshape %15 : (tensor<f32>) -> tensor<1xf32>
    %23 = stablehlo.reshape %19 : (tensor<f32>) -> tensor<1xf32>
    %24 = stablehlo.concatenate %21, %22, %23, dim = 0 : (tensor<1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<3xf32>
    %25 = stablehlo.constant dense<1.0> : tensor<3xf32>
    %26 = stablehlo.constant dense<0.0> : tensor<3xf32>
    %27 = stablehlo.constant dense<0> : tensor<i32>
    %30:2 = stablehlo.while(%28 = %27, %29 = %26) : tensor<i32>, tensor<3xf32>
    cond {
      %31 = stablehlo.constant dense<4> : tensor<i32>
      %32 = stablehlo.compare LT, %28, %31, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %32 : tensor<i1>
    } do {
      %33 = stablehlo.dynamic_slice %8, %28, sizes = [1] : (tensor<4xf32>, tensor<i32>) -> tensor<1xf32>
      %34 = stablehlo.reshape %33 : (tensor<1xf32>) -> tensor<f32>
      %35 = stablehlo.broadcast_in_dim %34, dims = [] : (tensor<f32>) -> tensor<3xf32>
      %36 = stablehlo.compare GE, %35, %20 : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xi1>
      %37 = stablehlo.compare LT, %35, %24 : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xi1>
      %38 = stablehlo.and %36, %37 : tensor<3xi1>
      %39 = stablehlo.select %38, %25, %26 : (tensor<3xi1>, tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
      %40 = stablehlo.add %29, %39 : tensor<3xf32>
      %41 = stablehlo.constant dense<1> : tensor<i32>
      %42 = stablehlo.add %28, %41 : tensor<i32>
      stablehlo.return %42, %40 : tensor<i32>, tensor<3xf32>
    }
    %43 = stablehlo.convert %30#1 : (tensor<3xf32>) -> tensor<3xi32>
    return %43, %2 : tensor<3xi32>, tensor<2xui64>
  }
}
