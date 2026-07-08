module {
  func.func @sample(%key: tensor<2xui64>) -> (tensor<4xf32>, tensor<2xui64>) {
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
    %16 = stablehlo.constant dense<0.0> : tensor<f32>
    %17 = stablehlo.constant dense<1.0> : tensor<f32>
    %18 = stablehlo.slice %6 [0:1] : (tensor<3xf32>) -> tensor<1xf32>
    %19 = stablehlo.reshape %18 : (tensor<1xf32>) -> tensor<f32>
    %20 = stablehlo.add %16, %19 : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %22 = stablehlo.compare LT, %21, %15 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
    %23 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %24 = stablehlo.broadcast_in_dim %7, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %25 = stablehlo.select %22, %23, %24 : (tensor<4xi1>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    %26 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %27 = stablehlo.add %26, %25 : tensor<4xf32>
    %28 = stablehlo.slice %6 [1:2] : (tensor<3xf32>) -> tensor<1xf32>
    %29 = stablehlo.reshape %28 : (tensor<1xf32>) -> tensor<f32>
    %30 = stablehlo.add %20, %29 : tensor<f32>
    %31 = stablehlo.broadcast_in_dim %30, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %32 = stablehlo.compare LT, %31, %15 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
    %33 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %34 = stablehlo.broadcast_in_dim %7, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %35 = stablehlo.select %32, %33, %34 : (tensor<4xi1>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    %36 = stablehlo.add %27, %35 : tensor<4xf32>
    return %36, %9 : tensor<4xf32>, tensor<2xui64>
  }
}
