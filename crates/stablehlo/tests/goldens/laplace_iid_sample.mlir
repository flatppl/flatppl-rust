module {
  func.func @sample(%key: tensor<2xui64>) -> (tensor<4xf32>, tensor<2xui64>) {
    %0 = stablehlo.constant dense<0.0> : tensor<f32>
    %1 = stablehlo.constant dense<1.0> : tensor<f32>
    %2 = stablehlo.constant dense<0.0> : tensor<f32>
    %3 = stablehlo.constant dense<1.0> : tensor<f32>
    %4, %5 = stablehlo.rng_bit_generator %key, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<4xui32>)
    %6 = stablehlo.constant dense<9> : tensor<4xui32>
    %7 = stablehlo.shift_right_logical %5, %6 : tensor<4xui32>
    %8 = stablehlo.convert %7 : (tensor<4xui32>) -> tensor<4xf32>
    %9 = stablehlo.constant dense<1.1920929E-7> : tensor<4xf32>
    %10 = stablehlo.multiply %8, %9 : tensor<4xf32>
    %11 = stablehlo.constant dense<0.5> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %13 = stablehlo.subtract %10, %12 : tensor<4xf32>
    %14 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %15 = stablehlo.compare GE, %13, %14 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
    %16 = stablehlo.constant dense<1.0> : tensor<f32>
    %17 = stablehlo.constant dense<-1.0> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %16, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %19 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %20 = stablehlo.select %15, %18, %19 : (tensor<4xi1>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    %21 = stablehlo.abs %13 : tensor<4xf32>
    %22 = stablehlo.constant dense<2.0> : tensor<f32>
    %23 = stablehlo.broadcast_in_dim %22, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %24 = stablehlo.multiply %23, %21 : tensor<4xf32>
    %25 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %26 = stablehlo.subtract %25, %24 : tensor<4xf32>
    %27 = stablehlo.log %26 : tensor<4xf32>
    %28 = stablehlo.multiply %20, %27 : tensor<4xf32>
    %29 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %30 = stablehlo.multiply %29, %28 : tensor<4xf32>
    %31 = stablehlo.negate %30 : tensor<4xf32>
    %32 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %33 = stablehlo.add %32, %31 : tensor<4xf32>
    return %33, %4 : tensor<4xf32>, tensor<2xui64>
  }
}
