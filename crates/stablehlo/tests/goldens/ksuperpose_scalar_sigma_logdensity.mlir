module {
  func.func @logdensity() -> tensor<f32> {
    %0 = stablehlo.constant dense<0.3> : tensor<f32>
    %1 = stablehlo.constant dense<1.2> : tensor<f32>
    %2 = stablehlo.reshape %0 : (tensor<f32>) -> tensor<1xf32>
    %3 = stablehlo.reshape %1 : (tensor<f32>) -> tensor<1xf32>
    %4 = stablehlo.concatenate %2, %3, dim = 0 : (tensor<1xf32>, tensor<1xf32>) -> tensor<2xf32>
    %5 = stablehlo.log %4 : tensor<2xf32>
    %6 = stablehlo.constant dense<0.5> : tensor<f32>
    %7 = stablehlo.constant dense<-1.0> : tensor<f32>
    %8 = stablehlo.constant dense<2.0> : tensor<f32>
    %9 = stablehlo.reshape %7 : (tensor<f32>) -> tensor<1xf32>
    %10 = stablehlo.reshape %8 : (tensor<f32>) -> tensor<1xf32>
    %11 = stablehlo.concatenate %9, %10, dim = 0 : (tensor<1xf32>, tensor<1xf32>) -> tensor<2xf32>
    %12 = stablehlo.constant dense<1.0> : tensor<f32>
    %13 = stablehlo.log %12 : tensor<f32>
    %14 = stablehlo.negate %13 : tensor<f32>
    %15 = stablehlo.constant dense<-0.9189385332046727> : tensor<f32>
    %16 = stablehlo.broadcast_in_dim %6, dims = [] : (tensor<f32>) -> tensor<2xf32>
    %17 = stablehlo.subtract %16, %11 : tensor<2xf32>
    %18 = stablehlo.broadcast_in_dim %12, dims = [] : (tensor<f32>) -> tensor<2xf32>
    %19 = stablehlo.divide %17, %18 : tensor<2xf32>
    %20 = stablehlo.constant dense<-0.5> : tensor<f32>
    %21 = stablehlo.multiply %19, %19 : tensor<2xf32>
    %22 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2xf32>
    %23 = stablehlo.multiply %22, %21 : tensor<2xf32>
    %24 = stablehlo.add %14, %15 : tensor<f32>
    %25 = stablehlo.broadcast_in_dim %24, dims = [] : (tensor<f32>) -> tensor<2xf32>
    %26 = stablehlo.add %25, %23 : tensor<2xf32>
    %27 = stablehlo.add %5, %26 : tensor<2xf32>
    %28 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %29 = stablehlo.reduce(%27 init: %28) applies stablehlo.maximum across dimensions = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
    %30 = stablehlo.broadcast_in_dim %29, dims = [] : (tensor<f32>) -> tensor<2xf32>
    %31 = stablehlo.subtract %27, %30 : tensor<2xf32>
    %32 = stablehlo.exponential %31 : tensor<2xf32>
    %33 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.reduce(%32 init: %33) applies stablehlo.add across dimensions = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
    %35 = stablehlo.log %34 : tensor<f32>
    %36 = stablehlo.add %35, %29 : tensor<f32>
    %37 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %38 = stablehlo.reduce(%4 init: %37) applies stablehlo.add across dimensions = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
    %39 = stablehlo.log %38 : tensor<f32>
    %40 = stablehlo.subtract %36, %39 : tensor<f32>
    return %40 : tensor<f32>
  }
}
