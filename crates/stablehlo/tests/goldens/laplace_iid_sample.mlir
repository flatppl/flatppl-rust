module {
  func.func @sample(%key: tensor<2xui64>) -> (tensor<4xf32>, tensor<2xui64>) {
    %0 = stablehlo.constant dense<0.0> : tensor<f32>
    %1 = stablehlo.constant dense<1.0> : tensor<f32>
    %2, %3 = stablehlo.rng_bit_generator %key, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<4xui32>)
    %4 = stablehlo.constant dense<9> : tensor<4xui32>
    %5 = stablehlo.shift_right_logical %3, %4 : tensor<4xui32>
    %6 = stablehlo.convert %5 : (tensor<4xui32>) -> tensor<4xf32>
    %7 = stablehlo.constant dense<1.1920929E-7> : tensor<4xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<4xf32>
    %9 = stablehlo.constant dense<0.5> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %11 = stablehlo.subtract %8, %10 : tensor<4xf32>
    %12 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %13 = stablehlo.compare GE, %11, %12 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
    %14 = stablehlo.constant dense<-1.0> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %16 = stablehlo.broadcast_in_dim %14, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %17 = stablehlo.select %13, %15, %16 : (tensor<4xi1>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    %18 = stablehlo.abs %11 : tensor<4xf32>
    %19 = stablehlo.constant dense<2.0> : tensor<f32>
    %20 = stablehlo.broadcast_in_dim %19, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %21 = stablehlo.multiply %20, %18 : tensor<4xf32>
    %22 = stablehlo.subtract %15, %21 : tensor<4xf32>
    %23 = stablehlo.log %22 : tensor<4xf32>
    %24 = stablehlo.multiply %17, %23 : tensor<4xf32>
    %25 = stablehlo.multiply %15, %24 : tensor<4xf32>
    %26 = stablehlo.negate %25 : tensor<4xf32>
    %27 = stablehlo.add %12, %26 : tensor<4xf32>
    return %27, %2 : tensor<4xf32>, tensor<2xui64>
  }
}
