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
    %14 = stablehlo.constant dense<0.5> : tensor<f32>
    %15 = stablehlo.reshape %13 : (tensor<f32>) -> tensor<1xf32>
    %16 = stablehlo.reshape %14 : (tensor<f32>) -> tensor<1xf32>
    %17 = stablehlo.concatenate %15, %16, dim = 0 : (tensor<1xf32>, tensor<1xf32>) -> tensor<2xf32>
    %18 = stablehlo.log %17 : tensor<2xf32>
    %19 = stablehlo.negate %18 : tensor<2xf32>
    %20 = stablehlo.constant dense<-0.9189385332046727> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %6, dims = [] : (tensor<f32>) -> tensor<2xf32>
    %22 = stablehlo.subtract %21, %12 : tensor<2xf32>
    %23 = stablehlo.divide %22, %17 : tensor<2xf32>
    %24 = stablehlo.constant dense<-0.5> : tensor<f32>
    %25 = stablehlo.multiply %23, %23 : tensor<2xf32>
    %26 = stablehlo.broadcast_in_dim %24, dims = [] : (tensor<f32>) -> tensor<2xf32>
    %27 = stablehlo.multiply %26, %25 : tensor<2xf32>
    %28 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2xf32>
    %29 = stablehlo.add %19, %28 : tensor<2xf32>
    %30 = stablehlo.add %29, %27 : tensor<2xf32>
    %31 = stablehlo.add %5, %30 : tensor<2xf32>
    %32 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %33 = stablehlo.reduce(%31 init: %32) applies stablehlo.maximum across dimensions = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
    %34 = stablehlo.broadcast_in_dim %33, dims = [] : (tensor<f32>) -> tensor<2xf32>
    %35 = stablehlo.subtract %31, %34 : tensor<2xf32>
    %36 = stablehlo.exponential %35 : tensor<2xf32>
    %37 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %38 = stablehlo.reduce(%36 init: %37) applies stablehlo.add across dimensions = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
    %39 = stablehlo.log %38 : tensor<f32>
    %40 = stablehlo.add %39, %33 : tensor<f32>
    %41 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %42 = stablehlo.reduce(%4 init: %41) applies stablehlo.add across dimensions = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
    %43 = stablehlo.log %42 : tensor<f32>
    %44 = stablehlo.subtract %40, %43 : tensor<f32>
    return %44 : tensor<f32>
  }
}
