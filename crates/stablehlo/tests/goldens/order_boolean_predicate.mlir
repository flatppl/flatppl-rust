module {
  func.func @logdensity(%arg0: tensor<4xf32>) -> (tensor<f32>, tensor<f32>, tensor<f32>) {
    %0 = stablehlo.constant dense<3.0> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %2 = stablehlo.compare GT, %arg0, %1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
    %3 = stablehlo.constant dense<false> : tensor<i1>
    %4 = stablehlo.reduce(%2 init: %3) applies stablehlo.or across dimensions = [0] : (tensor<4xi1>, tensor<i1>) -> tensor<i1>
    %5 = stablehlo.constant dense<1.0> : tensor<f32>
    %6 = stablehlo.constant dense<2.0> : tensor<f32>
    %7 = stablehlo.select %4, %5, %6 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %8 = stablehlo.constant dense<0.0> : tensor<f32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %10 = stablehlo.compare GT, %arg0, %9 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
    %11 = stablehlo.constant dense<true> : tensor<i1>
    %12 = stablehlo.reduce(%10 init: %11) applies stablehlo.and across dimensions = [0] : (tensor<4xi1>, tensor<i1>) -> tensor<i1>
    %13 = stablehlo.constant dense<1.0> : tensor<f32>
    %14 = stablehlo.constant dense<2.0> : tensor<f32>
    %15 = stablehlo.select %12, %13, %14 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %16 = stablehlo.constant dense<3.0> : tensor<f32>
    %17 = stablehlo.broadcast_in_dim %16, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %18 = stablehlo.compare GT, %arg0, %17 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
    %19 = stablehlo.constant dense<false> : tensor<i1>
    %20 = stablehlo.reduce(%18 init: %19) applies stablehlo.or across dimensions = [0] : (tensor<4xi1>, tensor<i1>) -> tensor<i1>
    %21 = stablehlo.constant dense<0.0> : tensor<f32>
    %22 = stablehlo.broadcast_in_dim %21, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %23 = stablehlo.compare GT, %arg0, %22 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
    %24 = stablehlo.constant dense<true> : tensor<i1>
    %25 = stablehlo.reduce(%23 init: %24) applies stablehlo.and across dimensions = [0] : (tensor<4xi1>, tensor<i1>) -> tensor<i1>
    %26 = stablehlo.and %20, %25 : tensor<i1>
    %27 = stablehlo.constant dense<1.0> : tensor<f32>
    %28 = stablehlo.constant dense<2.0> : tensor<f32>
    %29 = stablehlo.select %26, %27, %28 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    return %7, %15, %29 : tensor<f32>, tensor<f32>, tensor<f32>
  }
}
