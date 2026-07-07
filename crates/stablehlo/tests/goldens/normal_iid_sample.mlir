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
    %11 = stablehlo.constant dense<2.0> : tensor<4xf32>
    %12 = stablehlo.constant dense<1.0> : tensor<4xf32>
    %13 = stablehlo.multiply %10, %11 : tensor<4xf32>
    %14 = stablehlo.subtract %13, %12 : tensor<4xf32>
    %15 = chlo.erf_inv %14 : tensor<4xf32> -> tensor<4xf32>
    %16 = stablehlo.constant dense<1.4142135> : tensor<4xf32>
    %17 = stablehlo.multiply %15, %16 : tensor<4xf32>
    %18 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %19 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %20 = stablehlo.multiply %17, %18 : tensor<4xf32>
    %21 = stablehlo.add %20, %19 : tensor<4xf32>
    %22 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %23 = stablehlo.multiply %22, %21 : tensor<4xf32>
    %24 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %25 = stablehlo.add %24, %23 : tensor<4xf32>
    return %25, %4 : tensor<4xf32>, tensor<2xui64>
  }
}
