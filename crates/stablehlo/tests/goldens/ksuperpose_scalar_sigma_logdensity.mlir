module {
  func.func @logdensity() -> tensor<f32> {
    %0 = stablehlo.constant dense<0.3> : tensor<f32>
    %1 = stablehlo.constant dense<1.2> : tensor<f32>
    %2 = stablehlo.reshape %0 : (tensor<f32>) -> tensor<1xf32>
    %3 = stablehlo.reshape %1 : (tensor<f32>) -> tensor<1xf32>
    %4 = stablehlo.concatenate %2, %3, dim = 0 : (tensor<1xf32>, tensor<1xf32>) -> tensor<2xf32>
    %5 = stablehlo.log %4 : tensor<2xf32>
    %6 = stablehlo.constant dense<0.5> : tensor<f32>
    %7 = stablehlo.constant dense<1.0> : tensor<f32>
    %8 = stablehlo.negate %7 : tensor<f32>
    %9 = stablehlo.constant dense<2.0> : tensor<f32>
    %10 = stablehlo.reshape %8 : (tensor<f32>) -> tensor<1xf32>
    %11 = stablehlo.reshape %9 : (tensor<f32>) -> tensor<1xf32>
    %12 = stablehlo.concatenate %10, %11, dim = 0 : (tensor<1xf32>, tensor<1xf32>) -> tensor<2xf32>
    %13 = stablehlo.constant dense<1.0> : tensor<f32>
    %14 = stablehlo.log %13 : tensor<f32>
    %15 = stablehlo.negate %14 : tensor<f32>
    %16 = stablehlo.constant dense<-0.9189385332046727> : tensor<f32>
    %17 = stablehlo.broadcast_in_dim %6, dims = [] : (tensor<f32>) -> tensor<2xf32>
    %18 = stablehlo.subtract %17, %12 : tensor<2xf32>
    %19 = stablehlo.broadcast_in_dim %13, dims = [] : (tensor<f32>) -> tensor<2xf32>
    %20 = stablehlo.divide %18, %19 : tensor<2xf32>
    %21 = stablehlo.constant dense<-0.5> : tensor<f32>
    %22 = stablehlo.multiply %20, %20 : tensor<2xf32>
    %23 = stablehlo.broadcast_in_dim %21, dims = [] : (tensor<f32>) -> tensor<2xf32>
    %24 = stablehlo.multiply %23, %22 : tensor<2xf32>
    %25 = stablehlo.add %15, %16 : tensor<f32>
    %26 = stablehlo.broadcast_in_dim %25, dims = [] : (tensor<f32>) -> tensor<2xf32>
    %27 = stablehlo.add %26, %24 : tensor<2xf32>
    %28 = stablehlo.add %5, %27 : tensor<2xf32>
    %29 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %30 = stablehlo.reduce(%28 init: %29) applies stablehlo.maximum across dimensions = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
    %31 = stablehlo.broadcast_in_dim %30, dims = [] : (tensor<f32>) -> tensor<2xf32>
    %32 = stablehlo.subtract %28, %31 : tensor<2xf32>
    %33 = stablehlo.exponential %32 : tensor<2xf32>
    %34 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %35 = stablehlo.reduce(%33 init: %34) applies stablehlo.add across dimensions = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
    %36 = stablehlo.log %35 : tensor<f32>
    %37 = stablehlo.add %36, %30 : tensor<f32>
    %38 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %39 = stablehlo.reduce(%4 init: %38) applies stablehlo.add across dimensions = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
    %40 = stablehlo.log %39 : tensor<f32>
    %41 = stablehlo.subtract %37, %40 : tensor<f32>
    return %41 : tensor<f32>
  }
}
