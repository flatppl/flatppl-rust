module {
  func.func @sample(%key: tensor<2xui64>) -> (tensor<f32>, tensor<2xui64>) {
    %0 = stablehlo.constant dense<0.3> : tensor<f32>
    %1 = stablehlo.constant dense<0.0> : tensor<f32>
    %2 = stablehlo.constant dense<1.0> : tensor<f32>
    %3, %4 = stablehlo.rng_bit_generator %key, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<5xui32>)
    %5 = stablehlo.constant dense<9> : tensor<5xui32>
    %6 = stablehlo.shift_right_logical %4, %5 : tensor<5xui32>
    %7 = stablehlo.convert %6 : (tensor<5xui32>) -> tensor<5xf32>
    %8 = stablehlo.constant dense<1.1920929E-7> : tensor<5xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<5xf32>
    %10 = stablehlo.subtract %2, %1 : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [] : (tensor<f32>) -> tensor<5xf32>
    %12 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<f32>) -> tensor<5xf32>
    %13 = stablehlo.multiply %9, %11 : tensor<5xf32>
    %14 = stablehlo.add %13, %12 : tensor<5xf32>
    %15 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<5xf32>
    %16 = stablehlo.compare LT, %14, %15 : (tensor<5xf32>, tensor<5xf32>) -> tensor<5xi1>
    %17 = stablehlo.constant dense<1.0> : tensor<5xf32>
    %18 = stablehlo.constant dense<0.0> : tensor<5xf32>
    %19 = stablehlo.select %16, %17, %18 : (tensor<5xi1>, tensor<5xf32>, tensor<5xf32>) -> tensor<5xf32>
    %20 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %21 = stablehlo.reduce(%19 init: %20) applies stablehlo.add across dimensions = [0] : (tensor<5xf32>, tensor<f32>) -> tensor<f32>
    return %21, %3 : tensor<f32>, tensor<2xui64>
  }
}
