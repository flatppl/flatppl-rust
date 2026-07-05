module {
  func.func @sample() -> tensor<3xf32> {
    %0 = stablehlo.constant dense<0.2> : tensor<f32>
    %1 = stablehlo.constant dense<0.3> : tensor<f32>
    %2 = stablehlo.constant dense<0.5> : tensor<f32>
    %3 = stablehlo.reshape %0 : (tensor<f32>) -> tensor<1xf32>
    %4 = stablehlo.reshape %1 : (tensor<f32>) -> tensor<1xf32>
    %5 = stablehlo.reshape %2 : (tensor<f32>) -> tensor<1xf32>
    %6 = stablehlo.concatenate %3, %4, %5, dim = 0 : (tensor<1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<3xf32>
    %7 = stablehlo.constant dense<0.0> : tensor<f32>
    %8 = stablehlo.constant dense<1.0> : tensor<f32>
    %9 = stablehlo.constant dense<4> : tensor<1xi64>
    %10 = stablehlo.rng %7, %8, %9, distribution = UNIFORM : (tensor<f32>, tensor<f32>, tensor<1xi64>) -> tensor<4xf32>
    %11 = stablehlo.constant dense<0.0> : tensor<f32>
    %12 = stablehlo.slice %6 [0:1] : (tensor<3xf32>) -> tensor<1xf32>
    %13 = stablehlo.reshape %12 : (tensor<1xf32>) -> tensor<f32>
    %14 = stablehlo.add %11, %13 : tensor<f32>
    %15 = stablehlo.slice %6 [1:2] : (tensor<3xf32>) -> tensor<1xf32>
    %16 = stablehlo.reshape %15 : (tensor<1xf32>) -> tensor<f32>
    %17 = stablehlo.add %14, %16 : tensor<f32>
    %18 = stablehlo.slice %6 [2:3] : (tensor<3xf32>) -> tensor<1xf32>
    %19 = stablehlo.reshape %18 : (tensor<1xf32>) -> tensor<f32>
    %20 = stablehlo.add %17, %19 : tensor<f32>
    %21 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %22 = stablehlo.reshape %11 : (tensor<f32>) -> tensor<1xf32>
    %23 = stablehlo.reshape %14 : (tensor<f32>) -> tensor<1xf32>
    %24 = stablehlo.reshape %17 : (tensor<f32>) -> tensor<1xf32>
    %25 = stablehlo.concatenate %22, %23, %24, dim = 0 : (tensor<1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<3xf32>
    %26 = stablehlo.reshape %14 : (tensor<f32>) -> tensor<1xf32>
    %27 = stablehlo.reshape %17 : (tensor<f32>) -> tensor<1xf32>
    %28 = stablehlo.reshape %21 : (tensor<f32>) -> tensor<1xf32>
    %29 = stablehlo.concatenate %26, %27, %28, dim = 0 : (tensor<1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<3xf32>
    %30 = stablehlo.constant dense<1.0> : tensor<3xf32>
    %31 = stablehlo.constant dense<0.0> : tensor<3xf32>
    %32 = stablehlo.constant dense<0.0> : tensor<3xf32>
    %33 = stablehlo.constant dense<0> : tensor<i32>
    %36:2 = stablehlo.while(%34 = %33, %35 = %32) : tensor<i32>, tensor<3xf32>
    cond {
      %37 = stablehlo.constant dense<4> : tensor<i32>
      %38 = stablehlo.compare LT, %34, %37, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %38 : tensor<i1>
    } do {
      %39 = stablehlo.dynamic_slice %10, %34, sizes = [1] : (tensor<4xf32>, tensor<i32>) -> tensor<1xf32>
      %40 = stablehlo.reshape %39 : (tensor<1xf32>) -> tensor<f32>
      %41 = stablehlo.broadcast_in_dim %40, dims = [] : (tensor<f32>) -> tensor<3xf32>
      %42 = stablehlo.compare GE, %41, %25 : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xi1>
      %43 = stablehlo.compare LT, %41, %29 : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xi1>
      %44 = stablehlo.and %42, %43 : tensor<3xi1>
      %45 = stablehlo.select %44, %30, %31 : (tensor<3xi1>, tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
      %46 = stablehlo.add %35, %45 : tensor<3xf32>
      %47 = stablehlo.constant dense<1> : tensor<i32>
      %48 = stablehlo.add %34, %47 : tensor<i32>
      stablehlo.return %48, %46 : tensor<i32>, tensor<3xf32>
    }
    return %36#1 : tensor<3xf32>
  }
}
