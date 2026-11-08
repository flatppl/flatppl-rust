module {
  func.func @logdensity(%arg0: tensor<f32>, %arg1: tensor<2xf32>) -> tensor<f32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<2xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<2xf32>
    %2 = stablehlo.constant dense<1.0> : tensor<f32>
    %3 = stablehlo.reshape %arg0 : (tensor<f32>) -> tensor<1xf32>
    %4 = stablehlo.reshape %2 : (tensor<f32>) -> tensor<1xf32>
    %8 = stablehlo.broadcast_in_dim %3, dims = [0] : (tensor<1xf32>) -> tensor<2xf32>
    %9 = stablehlo.subtract %1, %8 : tensor<2xf32>
    %10 = stablehlo.broadcast_in_dim %4, dims = [0] : (tensor<1xf32>) -> tensor<2xf32>
    %11 = stablehlo.divide %9, %10 : tensor<2xf32>
    %12 = stablehlo.constant dense<-0.5> : tensor<f32>
    %13 = stablehlo.reshape %11 : (tensor<2xf32>) -> tensor<2x1xf32>
    %14 = stablehlo.dot_general %13, %13, batching_dims = [0] x [0], contracting_dims = [1] x [1], precision = [HIGHEST, HIGHEST] : (tensor<2x1xf32>, tensor<2x1xf32>) -> tensor<2xf32>
    %15 = stablehlo.broadcast_in_dim %12, dims = [] : (tensor<f32>) -> tensor<2xf32>
    %16 = stablehlo.multiply %15, %14 : tensor<2xf32>
    %18 = stablehlo.constant dense<"0x8E3F6BBF"> : tensor<1xf32>
    %19 = stablehlo.broadcast_in_dim %18, dims = [0] : (tensor<1xf32>) -> tensor<2xf32>
    %20 = stablehlo.add %19, %16 : tensor<2xf32>
    %21 = stablehlo.abs %1 : tensor<2xf32>
    %22 = stablehlo.constant dense<0x7F800000> : tensor<2xf32>
    %23 = stablehlo.compare EQ, %21, %22 : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
    %24 = stablehlo.constant dense<0xFF800000> : tensor<2xf32>
    %25 = stablehlo.select %23, %24, %20 : (tensor<2xi1>, tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
    %26 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %27 = stablehlo.reduce(%25 init: %26) applies stablehlo.add across dimensions = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
    return %27 : tensor<f32>
  }
}
