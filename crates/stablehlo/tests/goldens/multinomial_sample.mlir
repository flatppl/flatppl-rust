module {
  func.func @sample(%key: tensor<2xui64>) -> (tensor<3xf32>, tensor<2xui64>) {
    %0 = stablehlo.constant dense<0.2> : tensor<f32>
    %1 = stablehlo.constant dense<0.3> : tensor<f32>
    %2 = stablehlo.constant dense<0.5> : tensor<f32>
    %3 = stablehlo.reshape %0 : (tensor<f32>) -> tensor<1xf32>
    %4 = stablehlo.reshape %1 : (tensor<f32>) -> tensor<1xf32>
    %5 = stablehlo.reshape %2 : (tensor<f32>) -> tensor<1xf32>
    %6 = stablehlo.concatenate %3, %4, %5, dim = 0 : (tensor<1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<3xf32>
    %7, %8 = stablehlo.rng_bit_generator %key, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<4xui32>)
    %9 = stablehlo.constant dense<9> : tensor<4xui32>
    %10 = stablehlo.shift_right_logical %8, %9 : tensor<4xui32>
    %11 = stablehlo.convert %10 : (tensor<4xui32>) -> tensor<4xf32>
    %12 = stablehlo.constant dense<1.1920929E-7> : tensor<4xf32>
    %13 = stablehlo.multiply %11, %12 : tensor<4xf32>
    %14 = stablehlo.constant dense<0.0> : tensor<f32>
    %15 = stablehlo.slice %6 [0:1] : (tensor<3xf32>) -> tensor<1xf32>
    %16 = stablehlo.reshape %15 : (tensor<1xf32>) -> tensor<f32>
    %17 = stablehlo.add %14, %16 : tensor<f32>
    %18 = stablehlo.slice %6 [1:2] : (tensor<3xf32>) -> tensor<1xf32>
    %19 = stablehlo.reshape %18 : (tensor<1xf32>) -> tensor<f32>
    %20 = stablehlo.add %17, %19 : tensor<f32>
    %21 = stablehlo.slice %6 [2:3] : (tensor<3xf32>) -> tensor<1xf32>
    %22 = stablehlo.reshape %21 : (tensor<1xf32>) -> tensor<f32>
    %23 = stablehlo.add %20, %22 : tensor<f32>
    %24 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %25 = stablehlo.reshape %14 : (tensor<f32>) -> tensor<1xf32>
    %26 = stablehlo.reshape %17 : (tensor<f32>) -> tensor<1xf32>
    %27 = stablehlo.reshape %20 : (tensor<f32>) -> tensor<1xf32>
    %28 = stablehlo.concatenate %25, %26, %27, dim = 0 : (tensor<1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<3xf32>
    %29 = stablehlo.reshape %17 : (tensor<f32>) -> tensor<1xf32>
    %30 = stablehlo.reshape %20 : (tensor<f32>) -> tensor<1xf32>
    %31 = stablehlo.reshape %24 : (tensor<f32>) -> tensor<1xf32>
    %32 = stablehlo.concatenate %29, %30, %31, dim = 0 : (tensor<1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<3xf32>
    %33 = stablehlo.constant dense<1.0> : tensor<3xf32>
    %34 = stablehlo.constant dense<0.0> : tensor<3xf32>
    %35 = stablehlo.constant dense<0.0> : tensor<3xf32>
    %36 = stablehlo.constant dense<0> : tensor<i32>
    %39:2 = stablehlo.while(%37 = %36, %38 = %35) : tensor<i32>, tensor<3xf32>
    cond {
      %40 = stablehlo.constant dense<4> : tensor<i32>
      %41 = stablehlo.compare LT, %37, %40, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %41 : tensor<i1>
    } do {
      %42 = stablehlo.dynamic_slice %13, %37, sizes = [1] : (tensor<4xf32>, tensor<i32>) -> tensor<1xf32>
      %43 = stablehlo.reshape %42 : (tensor<1xf32>) -> tensor<f32>
      %44 = stablehlo.broadcast_in_dim %43, dims = [] : (tensor<f32>) -> tensor<3xf32>
      %45 = stablehlo.compare GE, %44, %28 : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xi1>
      %46 = stablehlo.compare LT, %44, %32 : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xi1>
      %47 = stablehlo.and %45, %46 : tensor<3xi1>
      %48 = stablehlo.select %47, %33, %34 : (tensor<3xi1>, tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
      %49 = stablehlo.add %38, %48 : tensor<3xf32>
      %50 = stablehlo.constant dense<1> : tensor<i32>
      %51 = stablehlo.add %37, %50 : tensor<i32>
      stablehlo.return %51, %49 : tensor<i32>, tensor<3xf32>
    }
    return %39#1, %7 : tensor<3xf32>, tensor<2xui64>
  }
}
