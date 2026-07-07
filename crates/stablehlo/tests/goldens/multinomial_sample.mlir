module {
  func.func @sample(%key: tensor<2xui64>) -> (tensor<3xf32>, tensor<2xui64>) {
    %0 = stablehlo.constant dense<0.2> : tensor<f32>
    %1 = stablehlo.constant dense<0.3> : tensor<f32>
    %2 = stablehlo.constant dense<0.5> : tensor<f32>
    %3 = stablehlo.reshape %0 : (tensor<f32>) -> tensor<1xf32>
    %4 = stablehlo.reshape %1 : (tensor<f32>) -> tensor<1xf32>
    %5 = stablehlo.reshape %2 : (tensor<f32>) -> tensor<1xf32>
    %6 = stablehlo.concatenate %3, %4, %5, dim = 0 : (tensor<1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<3xf32>
    %7 = stablehlo.constant dense<0.0> : tensor<f32>
    %8 = stablehlo.constant dense<1.0> : tensor<f32>
    %9, %10 = stablehlo.rng_bit_generator %key, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<4xui32>)
    %11 = stablehlo.constant dense<9> : tensor<4xui32>
    %12 = stablehlo.shift_right_logical %10, %11 : tensor<4xui32>
    %13 = stablehlo.convert %12 : (tensor<4xui32>) -> tensor<4xf32>
    %14 = stablehlo.constant dense<1.1920929E-7> : tensor<4xf32>
    %15 = stablehlo.multiply %13, %14 : tensor<4xf32>
    %16 = stablehlo.subtract %8, %7 : tensor<f32>
    %17 = stablehlo.broadcast_in_dim %16, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %18 = stablehlo.broadcast_in_dim %7, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %19 = stablehlo.multiply %15, %17 : tensor<4xf32>
    %20 = stablehlo.add %19, %18 : tensor<4xf32>
    %21 = stablehlo.constant dense<0.0> : tensor<f32>
    %22 = stablehlo.slice %6 [0:1] : (tensor<3xf32>) -> tensor<1xf32>
    %23 = stablehlo.reshape %22 : (tensor<1xf32>) -> tensor<f32>
    %24 = stablehlo.add %21, %23 : tensor<f32>
    %25 = stablehlo.slice %6 [1:2] : (tensor<3xf32>) -> tensor<1xf32>
    %26 = stablehlo.reshape %25 : (tensor<1xf32>) -> tensor<f32>
    %27 = stablehlo.add %24, %26 : tensor<f32>
    %28 = stablehlo.slice %6 [2:3] : (tensor<3xf32>) -> tensor<1xf32>
    %29 = stablehlo.reshape %28 : (tensor<1xf32>) -> tensor<f32>
    %30 = stablehlo.add %27, %29 : tensor<f32>
    %31 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %32 = stablehlo.reshape %21 : (tensor<f32>) -> tensor<1xf32>
    %33 = stablehlo.reshape %24 : (tensor<f32>) -> tensor<1xf32>
    %34 = stablehlo.reshape %27 : (tensor<f32>) -> tensor<1xf32>
    %35 = stablehlo.concatenate %32, %33, %34, dim = 0 : (tensor<1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<3xf32>
    %36 = stablehlo.reshape %24 : (tensor<f32>) -> tensor<1xf32>
    %37 = stablehlo.reshape %27 : (tensor<f32>) -> tensor<1xf32>
    %38 = stablehlo.reshape %31 : (tensor<f32>) -> tensor<1xf32>
    %39 = stablehlo.concatenate %36, %37, %38, dim = 0 : (tensor<1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<3xf32>
    %40 = stablehlo.constant dense<1.0> : tensor<3xf32>
    %41 = stablehlo.constant dense<0.0> : tensor<3xf32>
    %42 = stablehlo.constant dense<0.0> : tensor<3xf32>
    %43 = stablehlo.constant dense<0> : tensor<i32>
    %46:2 = stablehlo.while(%44 = %43, %45 = %42) : tensor<i32>, tensor<3xf32>
    cond {
      %47 = stablehlo.constant dense<4> : tensor<i32>
      %48 = stablehlo.compare LT, %44, %47, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %48 : tensor<i1>
    } do {
      %49 = stablehlo.dynamic_slice %20, %44, sizes = [1] : (tensor<4xf32>, tensor<i32>) -> tensor<1xf32>
      %50 = stablehlo.reshape %49 : (tensor<1xf32>) -> tensor<f32>
      %51 = stablehlo.broadcast_in_dim %50, dims = [] : (tensor<f32>) -> tensor<3xf32>
      %52 = stablehlo.compare GE, %51, %35 : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xi1>
      %53 = stablehlo.compare LT, %51, %39 : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xi1>
      %54 = stablehlo.and %52, %53 : tensor<3xi1>
      %55 = stablehlo.select %54, %40, %41 : (tensor<3xi1>, tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
      %56 = stablehlo.add %45, %55 : tensor<3xf32>
      %57 = stablehlo.constant dense<1> : tensor<i32>
      %58 = stablehlo.add %44, %57 : tensor<i32>
      stablehlo.return %58, %56 : tensor<i32>, tensor<3xf32>
    }
    return %46#1, %9 : tensor<3xf32>, tensor<2xui64>
  }
}
